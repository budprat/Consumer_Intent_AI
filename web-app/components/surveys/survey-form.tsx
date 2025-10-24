// web-app/components/surveys/survey-form.tsx
"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import * as z from "zod";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Switch } from "@/components/ui/switch";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useCreateSurvey } from "@/hooks/use-create-survey";
import { Loader2, ChevronLeft, ChevronRight } from "lucide-react";
import type { CreateSurveyRequest } from "@/lib/types";

// Zod validation schema
const surveySchema = z.object({
  product_name: z.string().min(1, "Product name is required").max(100),
  product_description: z
    .string()
    .min(10, "Description must be at least 10 characters")
    .max(500),
  cohort_size: z
    .number()
    .min(10, "Minimum 10 responses")
    .max(1000, "Maximum 1000 responses"),
  use_demographics: z.boolean(),
  demographic_condition: z
    .object({
      age_range: z.tuple([z.number(), z.number()]).optional(),
      gender: z.enum(["male", "female", "other", "any"]).optional(),
      income_bracket: z.enum(["low", "middle", "high", "any"]).optional(),
      location: z.string().optional(),
    })
    .optional(),
});

type SurveyFormData = z.infer<typeof surveySchema>;

const STEPS = [
  { id: 1, title: "Product Details", description: "Describe your product" },
  {
    id: 2,
    title: "Cohort Settings",
    description: "Configure survey parameters",
  },
  { id: 3, title: "Demographics", description: "Target specific audiences" },
];

export function SurveyForm() {
  const router = useRouter();
  const [currentStep, setCurrentStep] = useState(1);
  const { mutate: createSurvey, isPending, error } = useCreateSurvey();

  const {
    register,
    handleSubmit,
    watch,
    setValue,
    formState: { errors },
  } = useForm<SurveyFormData>({
    resolver: zodResolver(surveySchema),
    defaultValues: {
      product_name: "",
      product_description: "",
      cohort_size: 100,
      use_demographics: true,
      demographic_condition: {
        gender: "any",
        income_bracket: "any",
      },
    },
  });

  const useDemographics = watch("use_demographics");

  const onSubmit = (data: SurveyFormData) => {
    const payload: CreateSurveyRequest = {
      product_name: data.product_name,
      product_description: data.product_description,
      enable_demographics: data.use_demographics,
      enable_bias_detection: true,
      temperature: 1.0,
      averaging_strategy: "uniform",
      metadata: {
        cohort_size: data.cohort_size,
        demographic_filters: data.use_demographics
          ? {
              gender:
                data.demographic_condition?.gender !== "any"
                  ? data.demographic_condition?.gender
                  : undefined,
              income_bracket:
                data.demographic_condition?.income_bracket !== "any"
                  ? data.demographic_condition?.income_bracket
                  : undefined,
              location: data.demographic_condition?.location || undefined,
            }
          : undefined,
      },
    };

    createSurvey(payload, {
      onSuccess: (survey: { survey_id: string }) => {
        router.push(`/surveys/${survey.survey_id}`);
      },
    });
  };

  const nextStep = () =>
    setCurrentStep((prev) => Math.min(prev + 1, STEPS.length));
  const prevStep = () => setCurrentStep((prev) => Math.max(prev - 1, 1));

  return (
    <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
      {/* Progress Indicator */}
      <div className="flex items-center justify-between mb-8">
        {STEPS.map((step, index) => (
          <div key={step.id} className="flex items-center">
            <div
              className={`flex items-center justify-center w-10 h-10 rounded-full border-2 ${
                currentStep >= step.id
                  ? "border-blue-600 bg-blue-600 text-white"
                  : "border-slate-300 bg-white text-slate-400"
              }`}
            >
              {step.id}
            </div>
            {index < STEPS.length - 1 && (
              <div
                className={`h-1 w-20 mx-2 ${
                  currentStep > step.id ? "bg-blue-600" : "bg-slate-200"
                }`}
              />
            )}
          </div>
        ))}
      </div>

      {/* Step Content */}
      <Card>
        <CardHeader>
          <CardTitle>{STEPS[currentStep - 1].title}</CardTitle>
          <CardDescription>
            {STEPS[currentStep - 1].description}
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {currentStep === 1 && (
            <>
              <div className="space-y-2">
                <Label htmlFor="product_name">Product Name</Label>
                <Input
                  id="product_name"
                  {...register("product_name")}
                  placeholder="e.g., EcoBottle Pro"
                />
                {errors.product_name && (
                  <p className="text-sm text-red-500">
                    {errors.product_name.message}
                  </p>
                )}
              </div>

              <div className="space-y-2">
                <Label htmlFor="product_description">Product Description</Label>
                <Textarea
                  id="product_description"
                  {...register("product_description")}
                  placeholder="Describe your product's features and benefits..."
                  rows={5}
                />
                {errors.product_description && (
                  <p className="text-sm text-red-500">
                    {errors.product_description.message}
                  </p>
                )}
              </div>
            </>
          )}

          {currentStep === 2 && (
            <>
              <div className="space-y-2">
                <Label htmlFor="cohort_size">
                  Cohort Size (Number of Responses)
                </Label>
                <Input
                  id="cohort_size"
                  type="number"
                  {...register("cohort_size", { valueAsNumber: true })}
                  min={10}
                  max={1000}
                />
                {errors.cohort_size && (
                  <p className="text-sm text-red-500">
                    {errors.cohort_size.message}
                  </p>
                )}
                <p className="text-sm text-slate-500">
                  Larger cohorts provide more reliable results but take longer
                  to process
                </p>
              </div>
            </>
          )}

          {currentStep === 3 && (
            <>
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label htmlFor="use_demographics">
                    Enable Demographic Targeting
                  </Label>
                  <p className="text-sm text-slate-500">
                    Improve accuracy by targeting specific demographics (+40%
                    reliability)
                  </p>
                </div>
                <Switch
                  id="use_demographics"
                  checked={useDemographics}
                  onCheckedChange={(checked: boolean) =>
                    setValue("use_demographics", checked)
                  }
                />
              </div>

              {useDemographics && (
                <div className="space-y-4 pt-4 border-t">
                  <div className="space-y-2">
                    <Label htmlFor="gender">Gender</Label>
                    <Select
                      onValueChange={(value: string) =>
                        setValue(
                          "demographic_condition.gender",
                          value as "male" | "female" | "other" | "any",
                        )
                      }
                      defaultValue="any"
                    >
                      <SelectTrigger id="gender">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="any">Any</SelectItem>
                        <SelectItem value="male">Male</SelectItem>
                        <SelectItem value="female">Female</SelectItem>
                        <SelectItem value="other">Other</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="income_bracket">Income Bracket</Label>
                    <Select
                      onValueChange={(value: string) =>
                        setValue(
                          "demographic_condition.income_bracket",
                          value as "low" | "middle" | "high" | "any",
                        )
                      }
                      defaultValue="any"
                    >
                      <SelectTrigger id="income_bracket">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="any">Any</SelectItem>
                        <SelectItem value="low">Low Income</SelectItem>
                        <SelectItem value="middle">Middle Income</SelectItem>
                        <SelectItem value="high">High Income</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="location">Location (Optional)</Label>
                    <Input
                      id="location"
                      {...register("demographic_condition.location")}
                      placeholder="e.g., United States, California, etc."
                    />
                  </div>
                </div>
              )}
            </>
          )}
        </CardContent>
        <CardFooter className="flex justify-between">
          <Button
            type="button"
            variant="outline"
            onClick={prevStep}
            disabled={currentStep === 1 || isPending}
          >
            <ChevronLeft className="mr-2 h-4 w-4" />
            Previous
          </Button>

          {currentStep < STEPS.length ? (
            <Button type="button" onClick={nextStep}>
              Next
              <ChevronRight className="ml-2 h-4 w-4" />
            </Button>
          ) : (
            <Button type="submit" disabled={isPending}>
              {isPending && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
              Create Survey
            </Button>
          )}
        </CardFooter>
      </Card>

      {error && (
        <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded">
          <p className="font-medium">Failed to create survey</p>
          <p className="text-sm">{error.message}</p>
        </div>
      )}
    </form>
  );
}

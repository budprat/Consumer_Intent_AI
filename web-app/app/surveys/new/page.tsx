"use client";

import { SurveyForm } from "@/components/surveys/survey-form";
import { PageHeader } from "@/components/shared/page-header";

export default function NewSurveyPage() {
  return (
    <div className="space-y-6">
      <PageHeader
        title="Create New Survey"
        description="Configure your purchase intent survey"
      />
      <SurveyForm />
    </div>
  );
}
